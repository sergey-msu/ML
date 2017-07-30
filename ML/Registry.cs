using System.Collections.Generic;
using ML.Core.Kernels;
using ML.Core.Metric;
using ML.Core.Logical;
using ML.Core.ActivationFunctions;
using ML.Contracts;
using ML.Utils;
using ML.Core.Distributions;
using ML.TextMethods.WeightingSchemes;

namespace ML.Registry
{
  public static class Kernel
  {
    public static readonly GaussianKernel Gaussian = new GaussianKernel();
    public static readonly QuadraticKernel Quadratic = new QuadraticKernel();
    public static readonly QuarticKernel Quartic = new QuarticKernel();
    public static readonly RectangularKernel Rectangular = new RectangularKernel();
    public static readonly TriangularKernel Triangular = new TriangularKernel();

    public static readonly Dictionary<string, IKernel> ByID = new Dictionary<string, IKernel>
    {
      { Gaussian.Name,    Gaussian },
      { Quadratic.Name,   Quadratic },
      { Quartic.Name,     Quartic },
      { Rectangular.Name, Rectangular },
      { Triangular.Name,  Triangular }
    };
  }

  public static class Metric
  {
    private static readonly Dictionary<double, LpMetric> s_Lp = new Dictionary<double, LpMetric>();

    public static readonly EuclideanMetric Euclidean = new EuclideanMetric();
    public static readonly LInftyMetric LInfty = new LInftyMetric();

    public static readonly Dictionary<string, IMetric<double[]>> ByID = new Dictionary<string, IMetric<double[]>>
    {
      { Euclidean.Name, Euclidean },
      { LInfty.Name,    LInfty }
    };

    public static LpMetric Lp(double p)
    {
      return GeneralUtils.GetThroughMap(p, s_Lp);
    }
  }

  public static class Informativity<TObj>
  {
    public static readonly GiniIndex<TObj> Gini = new GiniIndex<TObj>();
    public static readonly DonskoyIndex<TObj> Donskoy = new DonskoyIndex<TObj>();
    public static readonly EntropyIndex<TObj> Entropy = new EntropyIndex<TObj>();

    public static readonly Dictionary<string, IInformativityIndex<TObj>> ByID = new Dictionary<string, IInformativityIndex<TObj>>
    {
      { Gini.Name,    Gini },
      { Donskoy.Name, Donskoy },
      { Entropy.Name, Entropy }
    };
  }

  public static class Activation
  {
    private static readonly Dictionary<double, RationalActivation>    s_Rationals = new Dictionary<double, RationalActivation>();
    private static readonly Dictionary<double, ShiftedStepActivation> s_ShiftedSteps = new Dictionary<double, ShiftedStepActivation>();
    private static readonly Dictionary<double, LogisticActivation>    s_Logistics = new Dictionary<double, LogisticActivation>();
    private static readonly Dictionary<double, LeakyReLUActivation>   s_LeakyReLUs = new Dictionary<double, LeakyReLUActivation>();

    public static readonly ArctanActivation Atan = new ArctanActivation();
    public static readonly StepActivation Step = new StepActivation();
    public static readonly IdentityActivation Identity = new IdentityActivation();
    public static readonly ReLUActivation ReLU = new ReLUActivation();
    public static readonly TanhActivation Tanh = new TanhActivation();
    public static readonly ExpActivation Exp = new ExpActivation();
    public static readonly SignActivation Sign = new SignActivation();

    public static readonly Dictionary<string, IFunction> ByID = new Dictionary<string, IFunction>
    {
      { Atan.Name,        Atan },
      { Step.Name,        Step },
      { Identity.Name,    Identity },
      { ReLU.Name,        ReLU },
      { LeakyReLU().Name, LeakyReLU() },
      { Tanh.Name,        Tanh },
      { Exp.Name,         Exp },
      { Sign.Name,        Sign }
    };

    public static ShiftedStepActivation ShiftedStep(double p)
    {
      return GeneralUtils.GetThroughMap(p, s_ShiftedSteps);
    }

    public static RationalActivation Rational(double p)
    {
      return GeneralUtils.GetThroughMap(p, s_Rationals);
    }

    public static LogisticActivation Logistic(double a)
    {
      return GeneralUtils.GetThroughMap(a, s_Logistics);
    }

    public static LeakyReLUActivation LeakyReLU(double leak = LeakyReLUActivation.DFT_LEAK)
    {
      return GeneralUtils.GetThroughMap(leak, s_LeakyReLUs);
    }
  }

  public static class Distribution
  {
    public static NormalDistribution Normal(double mu, double sigma)
    {
      var result = new NormalDistribution();
      result.Params = new NormalDistribution.Parameters(mu, sigma);

      return result;
    }

    public static BernoulliDistribution Bernoulli(double p)
    {
      var result = new BernoulliDistribution();
      result.Params = new BernoulliDistribution.Parameters(p);

      return result;
    }
  }

  public static class TFWeightingScheme
  {
    private static readonly Dictionary<double, DoubleNormalizationTFWeightingScheme> s_DoubleNormalizations = new Dictionary<double, DoubleNormalizationTFWeightingScheme>();

    public static readonly BinaryTFWeightingScheme Binary = new BinaryTFWeightingScheme();
    public static readonly RawCountTFWeightingScheme RawCount = new RawCountTFWeightingScheme();
    public static readonly LogNormalizationTFWeightingScheme LogNormalization = new LogNormalizationTFWeightingScheme();

    public static DoubleNormalizationTFWeightingScheme DoubleNormalization(double k=0.5)
    {
      return GeneralUtils.GetThroughMap(k, s_DoubleNormalizations);
    }

    public static readonly Dictionary<string, ITFWeightingScheme> ByID = new Dictionary<string, ITFWeightingScheme>
    {
      { Binary.Name,    Binary },
      { RawCount.Name,   RawCount },
      { LogNormalization.Name,     LogNormalization },
      { DoubleNormalization().Name, DoubleNormalization() }
    };
  }

  public static class IDFWeightingScheme
  {
    public static readonly UnaryIDFWeightingScheme Unary = new UnaryIDFWeightingScheme();
    public static readonly StandartIDFWeightingScheme Standart = new StandartIDFWeightingScheme();
    public static readonly MaxIDFWeightingScheme Max = new MaxIDFWeightingScheme();
    public static readonly SmoothIDFWeightingScheme Smooth = new SmoothIDFWeightingScheme();
    public static readonly ProbabilisticIDFWeightingScheme Probabilistic = new ProbabilisticIDFWeightingScheme();

    public static readonly Dictionary<string, IIDFWeightingScheme> ByID = new Dictionary<string, IIDFWeightingScheme>
    {
      { Unary.Name,         Unary },
      { Standart.Name,      Standart },
      { Max.Name,           Max },
      { Smooth.Name,        Smooth },
      { Probabilistic.Name, Probabilistic },
    };
  }
}


