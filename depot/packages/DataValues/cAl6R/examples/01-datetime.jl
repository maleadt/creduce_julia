using DataValues
using Dates

a, b = DataValue("14:00:00"), DataValue("15:15:00")

Dates.value(DateTime(a,"HH:MM:SS") - DateTime(b,"HH:MM:SS"))

c = DateTime(a,"HH:MM:SS") - DateTime(b,"HH:MM:SS")
