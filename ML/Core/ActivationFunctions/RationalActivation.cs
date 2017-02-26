using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Rational Activation Function
  /// </summary>
  public class RationalActivation : IFunction
  {
    private double m_Shift;

    public RationalActivation(double shift)
    {
      if (shift <= 0)
        throw new MLException("Shift must be positive");

      m_Shift = shift;
    }

    public string ID { get { return "RAT"; } }
    public string Name { get { return "Rational"; } }
    public double Shift { get { return m_Shift; } }


    public double Value(double r)
    {
      return r > 0 ?
             1 - m_Shift/(2*(r+m_Shift)) :
             m_Shift/(2*(m_Shift-r));
    }

    public double Derivative(double r)
    {
      var d = Math.Abs(r) + m_Shift;
      return m_Shift / (2*d*d);
    }
  }
}
