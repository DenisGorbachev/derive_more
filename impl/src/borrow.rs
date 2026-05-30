//! Implementation of a [`Borrow`] derive macro.

use proc_macro2::TokenStream;
use quote::{format_ident, quote, ToTokens as _};
use syn::{
    parse::{Parse, ParseStream},
    parse_quote,
    spanned::Spanned as _,
};

use crate::utils::{
    add_extra_generic_type_param,
    attr::{self, ParseMultiple as _},
    Either, GenericsSearch, Spanning,
};

/// Expands a [`Borrow`] derive macro.
pub(crate) fn expand(
    input: &syn::DeriveInput,
    trait_name: &'static str,
) -> syn::Result<TokenStream> {
    let trait_ident = format_ident!("{trait_name}");
    let attr_name = format_ident!("borrow");

    let data = match &input.data {
        syn::Data::Struct(data) => Ok(data),
        syn::Data::Enum(e) => Err(syn::Error::new(
            e.enum_token.span(),
            format!("`{trait_ident}` cannot be derived for enums"),
        )),
        syn::Data::Union(u) => Err(syn::Error::new(
            u.union_token.span(),
            format!("`{trait_ident}` cannot be derived for unions"),
        )),
    }?;

    if data.fields.len() != 1 {
        return Err(syn::Error::new(
            if data.fields.is_empty() {
                data.struct_token.span
            } else {
                data.fields.span()
            },
            format!("`{trait_ident}` can only be derived for structs with exactly one field"),
        ));
    }

    let field = data.fields.iter().next().unwrap();
    let struct_attr = StructAttribute::parse_attrs(&input.attrs, &attr_name)?;
    let field_attr = FieldAttribute::parse_attrs(&field.attrs, &attr_name)?;

    if struct_attr.is_some() && field_attr.is_some() {
        return Err(syn::Error::new(
            field.span(),
            format!(
                "`#[{attr_name}(...)]` cannot be placed on both struct and its field"
            ),
        ));
    }

    let is_forwarded = if struct_attr.is_some() {
        true
    } else if let Some(attr) = field_attr {
        match attr.item {
            FieldAttribute::Direct(_) => false,
            FieldAttribute::Forward(_) => true,
            FieldAttribute::Skip(skip) => {
                return Err(syn::Error::new(
                    attr.span,
                    format!(
                        "`#[{attr_name}({})]` cannot be used when deriving `{trait_ident}` \
                         because exactly one field must be borrowed",
                        skip.name(),
                    ),
                ));
            }
        }
    } else {
        false
    };

    validate_field_type(field, &input.generics, is_forwarded, &trait_ident)?;

    Ok(Expansion {
        ident: &input.ident,
        generics: &input.generics,
        field,
        field_index: 0,
        is_forwarded,
    }
    .into_token_stream())
}

fn validate_field_type(
    field: &syn::Field,
    generics: &syn::Generics,
    is_forwarded: bool,
    trait_ident: &syn::Ident,
) -> syn::Result<()> {
    let field_ty = &field.ty;

    if is_assoc_type_involving_generic_param(field_ty, generics) {
        return Err(syn::Error::new(
            field_ty.span(),
            format!(
                "`{trait_ident}` cannot be derived for an associated type projection involving \
                 generic parameters because the impl can overlap with `core`'s blanket `Borrow` \
                 implementation",
            ),
        ));
    }

    if is_forwarded && is_forwarded_overlap(field_ty, generics) {
        return Err(syn::Error::new(
            field_ty.span(),
            "`#[borrow(forward)]` cannot be used on a generic parameter field, an associated \
             type projection involving generic parameters, or a forwarding pointer to one \
             because the impl can overlap with `core`'s blanket `Borrow` implementation",
        ));
    }

    Ok(())
}

fn is_assoc_type_involving_generic_param(
    ty: &syn::Type,
    generics: &syn::Generics,
) -> bool {
    let ty = match ty {
        syn::Type::Group(ty) => {
            return is_assoc_type_involving_generic_param(&ty.elem, generics)
        }
        syn::Type::Paren(ty) => {
            return is_assoc_type_involving_generic_param(&ty.elem, generics)
        }
        ty => ty,
    };

    let syn::Type::Path(ty) = ty else {
        return false;
    };

    if ty.qself.is_some() {
        return GenericsSearch::from(generics).any_in(&syn::Type::Path(ty.clone()));
    }

    ty.path.segments.len() > 1
        && generics
            .type_params()
            .any(|param| ty.path.segments[0].ident == param.ident)
}

fn is_bare_generic_param(ty: &syn::Type, generics: &syn::Generics) -> bool {
    let syn::Type::Path(ty) = ty else {
        return false;
    };

    ty.qself.is_none()
        && ty.path.segments.len() == 1
        && generics
            .type_params()
            .any(|param| ty.path.segments[0].ident == param.ident)
}

fn is_forwarded_overlap(ty: &syn::Type, generics: &syn::Generics) -> bool {
    let ty = peel_forwarding_pointer(ty);
    is_bare_generic_param(ty, generics)
        || is_assoc_type_involving_generic_param(ty, generics)
}

fn peel_forwarding_pointer(ty: &syn::Type) -> &syn::Type {
    match ty {
        syn::Type::Group(ty) => peel_forwarding_pointer(&ty.elem),
        syn::Type::Paren(ty) => peel_forwarding_pointer(&ty.elem),
        syn::Type::Reference(ty) => peel_forwarding_pointer(&ty.elem),
        syn::Type::Path(type_path) => {
            fundamental_path_inner_type(type_path).map_or(ty, peel_forwarding_pointer)
        }
        _ => ty,
    }
}

fn fundamental_path_inner_type(ty: &syn::TypePath) -> Option<&syn::Type> {
    let segment = fundamental_path_segment(ty)?;

    let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };

    arguments.args.iter().find_map(|arg| {
        if let syn::GenericArgument::Type(ty) = arg {
            Some(ty)
        } else {
            None
        }
    })
}

fn fundamental_path_segment(ty: &syn::TypePath) -> Option<&syn::PathSegment> {
    if ty.qself.is_some() {
        return None;
    }

    let segments = ty.path.segments.iter().collect::<Vec<_>>();
    match segments.as_slice() {
        [segment] if segment.ident == "Box" || segment.ident == "Pin" => Some(segment),
        [std_or_alloc, boxed, segment]
            if (std_or_alloc.ident == "std" || std_or_alloc.ident == "alloc")
                && boxed.ident == "boxed"
                && segment.ident == "Box" =>
        {
            Some(segment)
        }
        [std_or_core, pin, segment]
            if (std_or_core.ident == "std" || std_or_core.ident == "core")
                && pin.ident == "pin"
                && segment.ident == "Pin" =>
        {
            Some(segment)
        }
        [derive_more, core, pin, segment]
            if derive_more.ident == "derive_more"
                && core.ident == "core"
                && pin.ident == "pin"
                && segment.ident == "Pin" =>
        {
            Some(segment)
        }
        _ => None,
    }
}

/// Expansion of a macro for generating a [`Borrow`] implementation for a single field of a struct.
struct Expansion<'a> {
    /// [`syn::Ident`] of the struct.
    ident: &'a syn::Ident,

    /// [`syn::Generics`] of the struct.
    generics: &'a syn::Generics,

    /// [`syn::Field`] of the struct.
    field: &'a syn::Field,

    /// Index of the [`syn::Field`].
    field_index: usize,

    /// Whether `Borrow::borrow()` should be forwarded to the field's `Borrow` implementation.
    is_forwarded: bool,
}

impl quote::ToTokens for Expansion<'_> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        let field_ty = &self.field.ty;
        let field_ident = self.field.ident.as_ref().map_or_else(
            || syn::Index::from(self.field_index).to_token_stream(),
            quote::ToTokens::to_token_stream,
        );
        let ty_ident = &self.ident;
        let (_, ty_gens, _) = self.generics.split_for_impl();
        let borrow_ty = extra_borrow_type_param(self.generics);

        let trait_ty = if self.is_forwarded {
            quote! { #borrow_ty }
        } else {
            quote! { #field_ty }
        };
        let return_ty = if self.is_forwarded {
            quote! { #trait_ty }
        } else {
            borrowed_type_for_return(field_ty)
        };
        let trait_path = quote! {
            derive_more::core::borrow::Borrow<#trait_ty>
        };

        let generics = if self.is_forwarded {
            let mut generics = add_extra_generic_type_param(
                self.generics,
                quote! { #borrow_ty: ?derive_more::core::marker::Sized },
            );
            generics.make_where_clause().predicates.push(parse_quote! {
                #field_ty: #trait_path
            });
            generics
        } else {
            self.generics.clone()
        };
        let (impl_gens, _, where_clause) = generics.split_for_impl();

        let body = if self.is_forwarded {
            quote! {
                <#field_ty as #trait_path>::borrow(&self.#field_ident)
            }
        } else {
            quote! {
                &self.#field_ident
            }
        };

        quote! {
            #[allow(deprecated)] // omit warnings on deprecated fields/variants
            #[allow(unreachable_code)] // omit warnings for `!` and other unreachable types
            #[automatically_derived]
            impl #impl_gens #trait_path for #ty_ident #ty_gens #where_clause {
                #[inline]
                fn borrow(&self) -> &#return_ty {
                    #body
                }
            }
        }
        .to_tokens(tokens);
    }
}

fn extra_borrow_type_param(generics: &syn::Generics) -> syn::Ident {
    let mut ident = format_ident!("__BorrowT");
    let mut index = 0;

    while generics.params.iter().any(|param| match param {
        syn::GenericParam::Type(param) => param.ident == ident,
        syn::GenericParam::Const(param) => param.ident == ident,
        syn::GenericParam::Lifetime(_) => false,
    }) {
        index += 1;
        ident = format_ident!("__BorrowT{index}");
    }

    ident
}

fn borrowed_type_for_return(ty: &syn::Type) -> TokenStream {
    match ty {
        syn::Type::Group(ty) => borrowed_type_for_return(&ty.elem),
        syn::Type::Paren(ty) => borrowed_type_for_return(&ty.elem),
        syn::Type::TraitObject(ty) => {
            let mut ty = ty.clone();
            if !ty
                .bounds
                .iter()
                .any(|bound| matches!(bound, syn::TypeParamBound::Lifetime(_)))
            {
                ty.bounds.push(parse_quote! { 'static });
            }

            quote! { (#ty) }
        }
        _ => quote! { #ty },
    }
}

type StructAttribute = attr::Forward;

#[derive(Clone, Copy)]
enum FieldAttribute {
    Direct(attr::Empty),
    Forward(attr::Forward),
    Skip(attr::Skip),
}

type UntypedFieldAttribute = Either<attr::Empty, Either<attr::Forward, attr::Skip>>;

impl From<UntypedFieldAttribute> for FieldAttribute {
    fn from(value: UntypedFieldAttribute) -> Self {
        match value {
            UntypedFieldAttribute::Left(empty) => Self::Direct(empty),
            UntypedFieldAttribute::Right(Either::Left(forward)) => {
                Self::Forward(forward)
            }
            UntypedFieldAttribute::Right(Either::Right(skip)) => Self::Skip(skip),
        }
    }
}

impl From<FieldAttribute> for UntypedFieldAttribute {
    fn from(value: FieldAttribute) -> Self {
        match value {
            FieldAttribute::Direct(empty) => Self::Left(empty),
            FieldAttribute::Forward(forward) => Self::Right(Either::Left(forward)),
            FieldAttribute::Skip(skip) => Self::Right(Either::Right(skip)),
        }
    }
}

impl Parse for FieldAttribute {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        mod ident {
            use syn::custom_keyword;

            custom_keyword!(forward);
            custom_keyword!(skip);
            custom_keyword!(ignore);
        }

        let ahead = input.lookahead1();

        if ahead.peek(ident::forward) {
            input.parse::<attr::Forward>().map(Self::Forward)
        } else if ahead.peek(ident::skip) || ahead.peek(ident::ignore) {
            input.parse::<attr::Skip>().map(Self::Skip)
        } else {
            Err(ahead.error())
        }
    }
}

impl attr::ParseMultiple for FieldAttribute {
    fn parse_attr_with<P: attr::Parser>(
        attr: &syn::Attribute,
        parser: &P,
    ) -> syn::Result<Self> {
        if matches!(attr.meta, syn::Meta::Path(_)) {
            Ok(Self::Direct(attr::Empty))
        } else {
            attr.parse_args_with(|ps: ParseStream<'_>| parser.parse(ps))
        }
    }

    fn merge_attrs(
        prev: Spanning<Self>,
        new: Spanning<Self>,
        name: &syn::Ident,
    ) -> syn::Result<Spanning<Self>> {
        UntypedFieldAttribute::merge_attrs(
            prev.map(Into::into),
            new.map(Into::into),
            name,
        )
        .map(|attr| attr.map(Self::from))
    }
}
