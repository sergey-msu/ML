using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Logistic Activation Function
  /// </summary>
  public sealed class LogisticActivation : IFunction
  {
    public string ID { get { return "LGS"; } }
    public string Name { get { return "Logistic"; } }

    public double Value(double r)
    {
      return 1.0D / (1.0D + Math.Exp(-r));
    }

    public double Derivative(double r)
    {
      var val = 1.0D / (1.0D + Math.Exp(-r));
      return val * (1.0F - val);
    }
  }
}
