struct DataValue;
end
for op in (:!, )
    @eval begin
        import .$(op)
        $op(DataValue) where {} = isna0 end
end
