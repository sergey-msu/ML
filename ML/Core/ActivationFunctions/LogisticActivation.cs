using System;
using ML.Contracts;

namespace ML.Core.ActivationFunctions
{
  /// <summary>
  /// Logistic Activation Function
  /// </summary>
  public sealed class LogisticActivation : IFunction
  {
    private double m_Alpha;

    public LogisticActivation(double alpha)
    {
      if (alpha <= 0)
        throw new MLException("Alpha must be positive");

      m_Alpha = alpha;
    }

    public string ID { get { return "LGS"; } }
    public string Name { get { return "Logistic"; } }
    public double Alpha { get { return m_Alpha; } }

    public double Value(double r)
    {
      return 1.0D / (1.0D + Math.Exp(-m_Alpha*r));
    }

    public double Derivative(double r)
    {
      var val = 1.0D / (1.0D + Math.Exp(-m_Alpha*r));
      return m_Alpha * val * (1.0F - val);
    }
  }
}
