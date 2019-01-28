using DataValues

A = [DataValue(9), DataValue(8), DataValue(15)]

map(i->isna(i) ? false : get(i) % 3 == 0, A)

f(i) = isna(i) ? false : get(i) % 3 == 0

f.(A)
