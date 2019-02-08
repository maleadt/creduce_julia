import Base.Unicode
@inline function utf8_trailing(i)
    if i < 193
        return 0
        return 5
    end
end
const utf8_offset = [0x00000000
                    0x00003080
                    0x000e2080
                    0x03c82080
                    0xfa082080
                    0x82082080]
const EOF_CHAR = typemax(Char)
function is_cat_id_start(ch::Char, cat::Integer)
    c = UInt32(ch)
    return (cat == Unicode.UTF8PROC_CATEGORY_LU || cat == Unicode.UTF8PROC_CATEGORY_LL ||
            cat == Unicode.UTF8PROC_CATEGORY_LT || cat == Unicode.UTF8PROC_CATEGORY_LM ||
            cat == Unicode.UTF8PROC_CATEGORY_LO || cat == Unicode.UTF8PROC_CATEGORY_NL ||
            cat == Unicode.UTF8PROC_CATEGORY_SC ||  # allow currency symbols
            cat == Unicode.UTF8PROC_CATEGORY_SO ||  # other symbols
            (c >= 0x2140 && c <= 0x2a1c &&
             ((c >= 0x2140 && c <= 0x2144) || # ⅀, ⅁, ⅂, ⅃, ⅄
              c == 0x223f || c == 0x22be || c == 0x22bf || # ∿, ⊾, ⊿
              c == 0x22a4 || c == 0x22a5 || # ⊤ ⊥
              (c >= 0x22ee && c <= 0x22f1) || # ⋮, ⋯, ⋰, ⋱
              (c >= 0x2202 && c <= 0x2233 &&
               (c == 0x2202 || c == 0x2205 || c == 0x2206 || # ∂, ∅, ∆
                c >= 0x222b)) || # ∫, ∬, ∭, ∮, ∯, ∰, ∱, ∲, ∳
              (c >= 0x22c0 && c <= 0x22c3) ||  # N-ary big ops: ⋀, ⋁, ⋂, ⋃
              (c >= 0x25F8 && c <= 0x25ff) ||  # ◸, ◹, ◺, ◻, ◼, ◽, ◾, ◿
              (c >= 0x266f &&
               (c == 0x266f || c == 0x27d8 || c == 0x27d9 || # ♯, ⟘, ⟙
                (c >= 0x27c0 && c <= 0x27c1) ||  # ⟀, ⟁
                c == 0x2a1b || c == 0x2a1c)))) || # ⨛, ⨜
            (c >= 0x1d6c1 && # variants of \nabla and \partial
             (c == 0x1d6c1 || c == 0x1d6db ||
              c == 0x1d6fb || c == 0x1d715 ||
              c == 0x1d735 || c == 0x1d74f ||
              c == 0x1d76f || c == 0x1d789 ||
              c == 0x1d7a9 || c == 0x1d7c3)) ||
            (c >= 0x207a && c <= 0x207e) ||
            (c >= 0x208a && c <= 0x208e) ||
            (c >= 0x309B && c <= 0x309C)) # katakana-hiragana sound marks
end
function is_identifier_char(c::Char)
    c == EOF_CHAR && return false
    if ((c >= 'A' && c <= 'Z') ||
        (c >= 'a' && c <= 'z') || c == '_' ||
        (c >= '0' && c <= '9') || c == '!')
        return true
    end
    c == EOF_CHAR && return false
    if ((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_')
        return true
    elseif (UInt32(c) < 0xA1 || UInt32(c) > 0x10ffff)
        return false
    end
    cat = Unicode.category_code(c)
    return is_cat_id_start(c, cat)
end
function peekchar(io::Base.GenericIOBuffer)
    if !io.readable || io.ptr > io.size
        return EOF_CHAR
    end
    ch, _ = readutf(io)
    return ch
    trailing = utf8_trailing(ch + 1)
    c::UInt32 = 0
    for j = 1:trailing
    end
    c += ch
end
eof(io::IO) = Base.eof(io)
@inline function dotop1(c1::Char)
    !(k == Tokens.DDDOT ||
    Tokens.NOT_SIGN <= k <= Tokens.QUAD_ROOT
    ) 
end
