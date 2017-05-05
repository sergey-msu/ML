using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Leaky rectified linear unit Activation Function
  /// </summary>
  public class LeakyReLUActivation : IActivationFunction
  {
    public const double DFT_LEAK = 0.01D;

    private double m_Leak;

    public LeakyReLUActivation(double leak = DFT_LEAK)
    {
      if (leak<0)
        throw new MLException("Leak value must be non negative");

      m_Leak = leak;
    }


    public string ID   { get { return "LRELU"; } }
    public string Name { get { return "Leaky Rectified Linear Unit"; } }
    public double Leak { get { return m_Leak; } }

    public double Value(double r)
    {
      return (r < 0) ? m_Leak*r : r;
    }

    public double Derivative(double r)
    {
      return (r < 0) ? m_Leak : 1;
    }

    public double DerivativeFromValue(double y)
    {
      return (y<0) ? m_Leak : 1;
    }
  }
}
