from pyseus import load

# Full paths do not work some times, windows issue or something else?
# Sometimes Vs Code needs a restart to proper load the pictures again
# There is a difference between / and \ separator, for \ if there is a 0 after it, before the string there must be a r -> r'.\asdf..'

load('./tests/samples/sample.h5')


