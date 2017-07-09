using System.Collections.Generic;
using ML.Core.Kernels;
using ML.Core.Metric;
using ML.Core.Logical;
using ML.Core.ActivationFunctions;
using ML.Contracts;
using ML.Utils;
using ML.Core.Distributions;

namespace ML.Core.Registry
{
  public static class Kernel
  {
    public static readonly GaussianKernel    Gaussian    = new GaussianKernel();
    public static readonly QuadraticKernel   Quadratic   = new QuadraticKernel();
    public static readonly QuarticKernel     Quartic     = new QuarticKernel();
    public static readonly RectangularKernel Rectangular = new RectangularKernel();
    public static readonly TriangularKernel  Triangular  = new TriangularKernel();

    public static readonly Dictionary<string, IKernel> ByID = new Dictionary<string, IKernel>
    {
      { Gaussian.ID,    Gaussian },
      { Quadratic.ID,   Quadratic },
      { Quartic.ID,     Quartic },
      { Rectangular.ID, Rectangular },
      { Triangular.ID,  Triangular }
    };
  }

  public static class Metric
  {
    private static readonly Dictionary<double, LpMetric> m_Lp = new Dictionary<double, LpMetric>();

    public static readonly EuclideanMetric Euclidean = new EuclideanMetric();
    public static readonly LInftyMetric    LInfty    = new LInftyMetric();

    public static readonly Dictionary<string, IMetric<double[]>> ByID = new Dictionary<string, IMetric<double[]>>
    {
      { Euclidean.ID, Euclidean },
      { LInfty.ID,    LInfty }
    };

    public static LpMetric Lp(double p)
    {
      return GeneralUtils.GetThroughMap(p, m_Lp);
    }
  }

  public static class Informativity<TObj>
  {
    public static readonly GiniIndex<TObj>    Gini    = new GiniIndex<TObj>();
    public static readonly DonskoyIndex<TObj> Donskoy = new DonskoyIndex<TObj>();
    public static readonly EntropyIndex<TObj> Entropy = new EntropyIndex<TObj>();

    public static readonly Dictionary<string, IInformativityIndex<TObj>> ByID = new Dictionary<string, IInformativityIndex<TObj>>
    {
      { Gini.ID,    Gini },
      { Donskoy.ID, Donskoy },
      { Entropy.ID, Entropy }
    };
  }

  public static class Activation
  {
    private static readonly Dictionary<double, RationalActivation>    m_Rationals    = new Dictionary<double, RationalActivation>();
    private static readonly Dictionary<double, ShiftedStepActivation> m_ShiftedSteps = new Dictionary<double, ShiftedStepActivation>();
    private static readonly Dictionary<double, LogisticActivation>    m_Logistics    = new Dictionary<double, LogisticActivation>();
    private static readonly Dictionary<double, LeakyReLUActivation>   m_LeakyReLUs   = new Dictionary<double, LeakyReLUActivation>();

    public static readonly ArctanActivation    Atan      = new ArctanActivation();
    public static readonly StepActivation      Step      = new StepActivation();
    public static readonly IdentityActivation  Identity  = new IdentityActivation();
    public static readonly ReLUActivation      ReLU      = new ReLUActivation();
    public static readonly TanhActivation      Tanh      = new TanhActivation();
    public static readonly ExpActivation       Exp       = new ExpActivation();
    public static readonly SignActivation      Sign      = new SignActivation();

    public static readonly Dictionary<string, IFunction> ByID = new Dictionary<string, IFunction>
    {
      { Atan.ID,        Atan },
      { Step.ID,        Step },
      { Identity.ID,    Identity },
      { ReLU.ID,        ReLU },
      { LeakyReLU().ID, LeakyReLU() },
      { Tanh.ID,        Tanh },
      { Exp.ID,         Exp },
      { Sign.ID,        Sign }
    };

    public static ShiftedStepActivation ShiftedStep(double p)
    {
      return GeneralUtils.GetThroughMap(p, m_ShiftedSteps);
    }

    public static RationalActivation Rational(double p)
    {
      return GeneralUtils.GetThroughMap(p, m_Rationals);
    }

    public static LogisticActivation Logistic(double a)
    {
      return GeneralUtils.GetThroughMap(a, m_Logistics);
    }

    public static LeakyReLUActivation LeakyReLU(double leak = LeakyReLUActivation.DFT_LEAK)
    {
      return GeneralUtils.GetThroughMap(leak, m_LeakyReLUs);
    }
  }

  public static class Distribution
  {
    public static NormalDistribution Normal(double theta, double sigma)
    {
      return new NormalDistribution(theta, sigma);
    }
  }
}


