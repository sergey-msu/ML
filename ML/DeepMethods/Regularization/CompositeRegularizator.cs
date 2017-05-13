using System;
using System.Linq;
using ML.Contracts;

namespace ML.DeepMethods.Regularization
{
  public class CompositeRegularizator : IRegularizator
  {
    private IRegularizator[] m_Regularizators;

    public CompositeRegularizator(params IRegularizator[] regularizators)
    {
      m_Regularizators = regularizators.ToArray();
    }

    public IRegularizator[] Regularizators { get { return m_Regularizators; } }


    public double Value(double[][] weights)
    {
      var result = 0.0D;
      var rlen = m_Regularizators.Length;

      for (int i=0; i<rlen; i++)
      {
        var r = m_Regularizators[i];
        result += r.Value(weights);
      }

      return result;
    }

    public void Apply(double[][] gradients, double[][] weights)
    {
      var rlen = m_Regularizators.Length;

      for (int i=0; i<rlen; i++)
      {
        var r = m_Regularizators[i];
        r.Apply(gradients, weights);
      }
    }
  }
}
