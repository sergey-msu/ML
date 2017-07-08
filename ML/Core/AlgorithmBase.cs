using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using ML.Contracts;
using ML.Core.Mathematics;
using ML.Utils;

namespace ML.Core
{
  /// <summary>
  /// Base class for supervised algorithm with training sample
  /// </summary>
  public abstract class SupervisedAlgorithmBase<TSample, TObj, TMark> : ISupervisedAlgorithm<TSample, TObj, TMark>
      where TSample: MarkedSample<TObj, TMark>
  {
    private object m_ErrSynk = new object();
    private TSample m_TrainingSample;

    protected SupervisedAlgorithmBase()
    {
    }

    /// <summary>
    /// Algorithm mnemonic ID
    /// </summary>
    public abstract string ID { get; }

    /// <summary>
    /// Algorithm name
    /// </summary>
    public abstract string Name { get; }

    /// <summary>
    /// Training sample
    /// </summary>
    public TSample TrainingSample
    {
      get { return m_TrainingSample; }
      set
      {
        if (value == null || !value.Any())
        throw new MLException("SupervisedAlgorithmBase.ctor(trainingSample=null|empty)");

        m_TrainingSample = value;
      }
    }

    /// <summary>
    /// Train the algorithm with some initial marked data
    /// </summary>
    public void Train(TSample trainingSample)
    {
      TrainingSample = trainingSample;
      DoTrain();
    }

    /// <summary>
    /// Make a prediction
    /// </summary>
    public abstract TMark Predict(TObj obj);

    /// <summary>
    /// Returns all errors of the algorithm on some test classified sample
    /// </summary>
    public virtual IEnumerable<ErrorInfo<TObj, TMark>> GetErrors(TSample testSample, double threshold, bool parallel)
    {
      var errors = new List<ErrorInfo<TObj, TMark>>();
      var body = new Action<KeyValuePair<TObj, TMark>>(pdata =>
      {
        var predMark = this.Predict(pdata.Key);
        var realMark = pdata.Value;
        var proximity = CalculateProximity(predMark, realMark, threshold);
        if (proximity > 0)
        {
          lock (m_ErrSynk)
            errors.Add(new ErrorInfo<TObj, TMark>(pdata.Key, realMark, predMark));
         }
      });

      if (parallel)
        Parallel.ForEach(testSample, body);
      else
        foreach (var pdata in testSample)
          body(pdata);

      return errors;
    }


    protected abstract double CalculateProximity(TMark mark1, TMark mark2, double threshold);

    protected abstract void DoTrain();
  }

  /// <summary>
  /// Base class for algorithm that accepts whole traing sample
  /// </summary>
  public abstract class ClassificationAlgorithmBase<TObj>
    : SupervisedAlgorithmBase<ClassifiedSample<TObj>, TObj, Class>,
      IClassificationAlgorithm<TObj>
  {
    protected ClassificationAlgorithmBase()
    {
    }


    protected override double CalculateProximity(Class mark1, Class mark2, double threshold)
    {
      return (mark1 == mark2) ? 0 : 1;
    }
  }

  /// <summary>
  /// Base class for supervised algorithm for regression purposes
  /// </summary>
  public abstract class RegressionAlgorithmBase<TObj>
    : SupervisedAlgorithmBase<RegressionSample<TObj>, TObj, double>,
      IRegressionAlgorithm<TObj>
  {
    protected RegressionAlgorithmBase()
    {
    }


    protected override double CalculateProximity(double mark1, double mark2, double threshold)
    {
      var marg1 = threshold - mark1;
      var marg2 = threshold - mark2;

      if ((marg1>0 && marg2>0) || (marg1<=0 && marg2<=0)) return 0;
      return 1;
    }
  }

  /// <summary>
  /// Base class for supervised algorithm for multidimensional regression purposes
  /// </summary>
  public abstract class MultiRegressionAlgorithmBase<TObj>
    : SupervisedAlgorithmBase<MultiRegressionSample<TObj>, TObj, double[]>,
      IMultiRegressionAlgorithm<TObj>
  {
    protected MultiRegressionAlgorithmBase()
    {
    }


    protected override double CalculateProximity(double[] mark1, double[] mark2, double threshold)
    {
      if (mark1==null || mark2==null)
        return (mark1==mark2) ? 0 : double.PositiveInfinity;

      var len = mark1.Length;
      if (mark1.Length != mark2.Length) return double.PositiveInfinity;

      return GeneralUtils.ArgMax(mark1)==GeneralUtils.ArgMax(mark2) ? 0 : 1;
    }
  }
}
