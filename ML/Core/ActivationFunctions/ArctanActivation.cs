using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// ArcTangent Activation Function
  /// </summary>
  public class ArctanActivation : IFunction
  {
    public string ID { get { return "ATAN"; } }

    public string Name { get { return "ArcTangent Activation Function"; } }

    public double Value(double r)
    {
      return Math.Atan(r);
    }

    public double Derivative(double r)
    {
      return 1.0D / (1.0D + r*r);
    }
  }
}
