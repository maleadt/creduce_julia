@inline function utf8_trailing(i)
    if i < 193
        return 0
    end
end
const utf8_offset = [0x00000000
                    0x82082080]
function is_cat_id_start(ch::Char, cat::Integer)
    return (cat == Unicode.UTF8PROC_CATEGORY_LU || cat == Unicode.UTF8PROC_CATEGORY_LL ||
            cat == Unicode.UTF8PROC_CATEGORY_LT || cat == Unicode.UTF8PROC_CATEGORY_LM ||
            cat == Unicode.UTF8PROC_CATEGORY_LO || cat == Unicode.UTF8PROC_CATEGORY_NL ||
            (c >= 0x2140 && c <= 0x2a1c &&
             ((c >= 0x2140 && c <= 0x2144) || # ⅀, ⅁, ⅂, ⅃, ⅄
              (c >= 0x2202 && c <= 0x2233 &&
               (c == 0x2202 || c == 0x2205 || c == 0x2206 || # ∂, ∅, ∆
                c >= 0x222b)) || # ∫, ∬, ∭, ∮, ∯, ∰, ∱, ∲, ∳
              (c >= 0x266f &&
               (c == 0x266f || c == 0x27d8 || c == 0x27d9 || # ♯, ⟘, ⟙
                c == 0x2a1b || c == 0x2a1c)))) || # ⨛, ⨜
            (c >= 0x1d6c1 && # variants of \nabla and \partial
             (c == 0x1d6c1 || c == 0x1d6db ||
              c == 0x1d7a9 || c == 0x1d7c3)) ||
            (c >= 0x309B && c <= 0x309C)) # katakana-hiragana sound marks
end
function is_identifier_char(c::Char)
    c == EOF_CHAR && return false
    if ((c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '!')
    end
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_')
        return true
    elseif (UInt32(c) < 0xA1 || UInt32(c) > 0x10ffff)
        return false
    end
end
function peekchar(io::Base.GenericIOBuffer)
    if !io.readable || io.ptr > io.size
    end
    for j = 1:trailing
    end
end
eof(io::IO) = Base.eof(io)
@inline function dotop1(c1::Char)
    !(k == Tokens.DDDOT ||
    Tokens.NOT_SIGN <= k <= Tokens.QUAD_ROOT
    ) 
end
