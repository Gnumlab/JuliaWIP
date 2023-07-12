cd(@__DIR__)
using Pkg
Pkg.activate(".")


using ForwardDiff, FiniteDiff
f(x) = 2x^2 + x
ForwardDiff.derivative(f, 2.0)
FiniteDiff.finite_difference_derivative(f,2.0)

using PkgTemplates
t = Template(user="luca")
t("Demo1")