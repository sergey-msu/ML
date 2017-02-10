using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Rectified linear unit Activation Function
  /// </summary>
  public class ReLUActivation : IFunction
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
  }
}
