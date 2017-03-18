using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Rectified linear unit Activation Function
  /// </summary>
  public class ReLUActivation : IActivationFunction
  {
    public string ID { get { return "RELU"; } }
    public string Name { get { return "Rectified Linear Unit"; } }

    public double Value(double r)
    {
      return (r < 0) ? 0 : r;
    }

    public double Derivative(double r)
    {
      return (r < 0) ? 0 : 1;
    }

    public double DerivativeFromValue(double y)
    {
      return (y==0) ? 0 : 1;
    }
  }
}
