using ML.DeepMethods.LossFunctions;
using System.Collections.Generic;

namespace ML.DeepMethods
{
  public static class Loss
  {
    private static readonly Dictionary<double, LpLoss> m_Lp = new Dictionary<double, LpLoss>();

    public static readonly EuclideanLoss           Euclidean           = new EuclideanLoss();
    public static readonly CrossEntropyLoss        CrossEntropy        = new CrossEntropyLoss();
    public static readonly CrossEntropySoftMaxLoss CrossEntropySoftMax = new CrossEntropySoftMaxLoss();

    public static LpLoss Lp(double p)
    {
      LpLoss result;
      if (!m_Lp.TryGetValue(p, out result))
      {
        result = new LpLoss(p);
        m_Lp[p] = result;
      }

      return result;
    }
  }
}


