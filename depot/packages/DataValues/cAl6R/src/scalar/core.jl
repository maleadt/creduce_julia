struct DataValue
end
for op in (:!, )
    @eval begin
        import .$op
        $op(DataValue)   = isna0 end
end
