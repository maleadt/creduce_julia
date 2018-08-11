bar(x::SubArray) = x[] = 42

@generated foo() = bar(whatever)

macro unused() end

foo()
