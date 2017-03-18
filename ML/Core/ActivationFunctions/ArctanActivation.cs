using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// ArcTangent Activation Function
  /// </summary>
  public class ArctanActivation : IActivationFunction
  {
    public string ID { get { return "ATAN"; } }
    public string Name { get { return "Arctangent"; } }

    public double Value(double r)
    {
      return Math.Atan(r);
    }

    public double Derivative(double r)
    {
      return 1.0D / (1.0D + r*r);
    }

    public double DerivativeFromValue(double y)
    {
      var tan = Math.Tan(y);
      return 1.0D / (1.0D + tan*tan);
    }
  }
}
