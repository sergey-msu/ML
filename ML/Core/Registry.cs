using System.Collections.Generic;
using ML.Core.Kernels;
using ML.Core.Metric;
using ML.Core.Logical;

namespace ML.Core
{
  public static class Registry
  {
    public static class Kernels
    {
      private static readonly GaussianKernel    m_GaussianKernel    = new GaussianKernel();
      private static readonly QuadraticKernel   m_QuadraticKernel   = new QuadraticKernel();
      private static readonly QuarticKernel     m_QuarticKernel     = new QuarticKernel();
      private static readonly RectangularKernel m_RectangularKernel = new RectangularKernel();
      private static readonly TriangularKernel  m_TriangularKernel  = new TriangularKernel();

      public static GaussianKernel    GaussianKernel    { get { return m_GaussianKernel; } }
      public static QuadraticKernel   QuadraticKernel   { get { return m_QuadraticKernel; } }
      public static QuarticKernel     QuarticKernel     { get { return m_QuarticKernel; } }
      public static RectangularKernel RectangularKernel { get { return m_RectangularKernel; } }
      public static TriangularKernel  TriangularKernel  { get { return m_TriangularKernel; } }
    }

    public static class Metrics
    {
      private static readonly EuclideanMetric m_EuclideanMetric = new EuclideanMetric();
      private static readonly LInftyMetric    m_LInftyMetric    = new LInftyMetric();
      private static readonly Dictionary<float, LpMetric> m_LpMetrics= new Dictionary<float, LpMetric>();

      public static EuclideanMetric EuclideanMetric { get { return m_EuclideanMetric; } }
      public static LInftyMetric LInftyMetric { get { return m_LInftyMetric; } }
      public static LpMetric LpMetric(float p)
      {
        LpMetric result;
        if (!m_LpMetrics.TryGetValue(p, out result))
        {
          result = new LpMetric(p);
          m_LpMetrics[p] = result;
        }

        return result;
      }
    }

    public static class Informativities
    {
      private static readonly GiniInformativity m_GiniInformativity = new GiniInformativity();
      private static readonly DonskoyInformativity m_DonskoyInformativity = new DonskoyInformativity();
      private static readonly EntropyInformativity m_EntropyInformativity = new EntropyInformativity();

      public static GiniInformativity GiniInfomativity { get { return m_GiniInformativity; } }
      public static DonskoyInformativity DonskoyInformativity { get { return m_DonskoyInformativity; } }
      public static EntropyInformativity EntropyInformativity { get { return m_EntropyInformativity; } }
    }
  }
}
