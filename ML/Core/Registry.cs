using System.Collections.Generic;
using ML.Core.Kernels;
using ML.Core.Metric;
using ML.Core.Logical;
using ML.Core.ActivationFunctions;
using ML.Contracts;

namespace ML.Core
{
  public static class Registry
  {
    public static class Kernels
    {
      public static readonly GaussianKernel    GaussianKernel    = new GaussianKernel();
      public static readonly QuadraticKernel   QuadraticKernel   = new QuadraticKernel();
      public static readonly QuarticKernel     QuarticKernel     = new QuarticKernel();
      public static readonly RectangularKernel RectangularKernel = new RectangularKernel();
      public static readonly TriangularKernel  TriangularKernel  = new TriangularKernel();

      public static readonly Dictionary<string, IFunction> ByID = new Dictionary<string, IFunction>
      {
        { GaussianKernel.ID,    GaussianKernel },
        { QuadraticKernel.ID,   QuadraticKernel },
        { QuarticKernel.ID,     QuarticKernel },
        { RectangularKernel.ID, RectangularKernel },
        { TriangularKernel.ID,  TriangularKernel }
      };
    }

    public static class Metrics
    {
      private static readonly Dictionary<double, LpMetric> m_LpMetrics = new Dictionary<double, LpMetric>();

      public static readonly EuclideanMetric EuclideanMetric = new EuclideanMetric();
      public static readonly LInftyMetric    LInftyMetric    = new LInftyMetric();

      public static readonly Dictionary<string, IMetric> ByID = new Dictionary<string, IMetric>
      {
        { EuclideanMetric.ID, EuclideanMetric },
        { LInftyMetric.ID,    LInftyMetric }
      };

      public static LpMetric LpMetric(double p)
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

    public static class Informativities<TObj>
    {
      public static readonly GiniIndex<TObj>    GiniInformativity    = new GiniIndex<TObj>();
      public static readonly DonskoyIndex<TObj> DonskoyInformativity = new DonskoyIndex<TObj>();
      public static readonly EntropyIndex<TObj> EntropyInformativity = new EntropyIndex<TObj>();

      public static readonly Dictionary<string, IInformIndex<TObj>> ByID = new Dictionary<string, IInformIndex<TObj>>
      {
        { GiniInformativity.ID,    GiniInformativity },
        { DonskoyInformativity.ID, DonskoyInformativity },
        { EntropyInformativity.ID, EntropyInformativity }
      };
    }

    public static class ActivationFunctions
    {
      private static readonly Dictionary<double, RationalActivation>    m_Rationals    = new Dictionary<double, RationalActivation>();
      private static readonly Dictionary<double, ShiftedStepActivation> m_ShiftedSteps = new Dictionary<double, ShiftedStepActivation>();
      private static readonly Dictionary<double, LogisticActivation>    m_Logistics    = new Dictionary<double, LogisticActivation>();

      public static readonly ArctanActivation   Atan       = new ArctanActivation();
      public static readonly StepActivation     Step       = new StepActivation();
      public static readonly IdentityActivation Identity   = new IdentityActivation();
      public static readonly ReLUActivation     ReLU       = new ReLUActivation();
      public static readonly TanhActivation     Tanh       = new TanhActivation();
      public static readonly ExpActivation      Exp        = new ExpActivation();
      public static readonly SignActivation     Sign       = new SignActivation();

      public static readonly Dictionary<string, IFunction> ByID = new Dictionary<string, IFunction>
      {
        { Atan.ID,     Atan },
        { Step.ID,     Step },
        { Identity.ID, Identity },
        { ReLU.ID,     ReLU },
        { Tanh.ID,     Tanh },
        { Exp.ID,      Exp },
        { Sign.ID,     Sign }
      };

      public static ShiftedStepActivation ShiftedStep(double p)
      {
        ShiftedStepActivation result;
        if (!m_ShiftedSteps.TryGetValue(p, out result))
        {
          result = new ShiftedStepActivation(p);
          m_ShiftedSteps[p] = result;
        }

        return result;
      }

      public static RationalActivation Rational(double p)
      {
        RationalActivation result;
        if (!m_Rationals.TryGetValue(p, out result))
        {
          result = new RationalActivation(p);
          m_Rationals[p] = result;
        }

        return result;
      }

      public static LogisticActivation Logistic(double a)
      {
        LogisticActivation result;
        if (!m_Logistics.TryGetValue(a, out result))
        {
          result = new LogisticActivation(a);
          m_Logistics[a] = result;
        }

        return result;
      }
    }
  }
}
