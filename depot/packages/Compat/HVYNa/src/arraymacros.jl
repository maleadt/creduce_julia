macro dotcompat(x)
    esc(_compat(Base.Broadcast.__dot__(x)))
end
export @dotcompat
