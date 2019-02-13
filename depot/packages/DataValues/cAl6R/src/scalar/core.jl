struct a
end
for b in (:!, )
    @eval begin
        import .$b
        $b(a)   = c end
end
