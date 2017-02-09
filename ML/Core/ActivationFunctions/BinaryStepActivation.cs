using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Binary Step Activation Function
  /// </summary>
  public sealed class BinaryStepActivation : IFunction
  {
    public string ID { get { return "STEP"; } }

    public string Name { get { return "Binary Step Activation Function"; } }

    public double Value(double r)
    {
      return r < 0 ? 0 : 1;
    }

    public double Derivative(double r)
    {
      return 0;
    }
  }
}
