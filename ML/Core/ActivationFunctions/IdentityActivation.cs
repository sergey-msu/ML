using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Identity Activation Function
  /// </summary>
  public sealed class IdentityActivation : IFunction
  {
    public string ID { get { return "IDT"; } }

    public string Name { get { return "Identity Activation Function"; } }

    public double Value(double r)
    {
      return r;
    }

    public double Derivative(double r)
    {
      return 1.0D;
    }
  }
}
