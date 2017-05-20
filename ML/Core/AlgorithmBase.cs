using System;
using System.Linq;
using System.Collections.Generic;
using System.Threading.Tasks;
using ML.Contracts;
using ML.Core.Mathematics;

namespace ML.Core
{
  /// <summary>
  /// Base class for supervised algorithm with training sample
  /// </summary>
  public abstract class SupervisedAlgorithmBase<TSample, TObj, TMark>
    : ISupervisedAlgorithm<TSample, TObj, TMark>
      where TSample: MarkedSample<TObj, TMark>
  {
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
      internal set
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
    public virtual IEnumerable<ErrorInfo<TObj, TMark>> GetErrors(TSample testSample)
    {
      var errors = new List<ErrorInfo<TObj, TMark>>();
      Parallel.ForEach(testSample, pdata =>
      {
        var predMark = this.Predict(pdata.Key);
        var realMark = pdata.Value;
        if (!object.Equals(predMark, realMark))
          lock (errors)
          {
            errors.Add(new ErrorInfo<TObj, TMark>(pdata.Key, realMark, predMark));
          }
      });

      return errors;
    }

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

    public virtual Class Classify(TObj x, Class[] classes, double threshold = 0)
    {
      var result = Predict(x);
      return mapValueToClass(result, classes, threshold);
    }


    /// <summary>
    /// Returns all errors of the algorithm on some test classified sample
    /// </summary>
    public virtual IEnumerable<ErrorInfo<TObj, Class>> GetClassificationErrors(RegressionSample<TObj> testSample, Class[] classes, double threshold = 0)
    {
      var errors = new List<ErrorInfo<TObj, Class>>();
      Parallel.ForEach(testSample, pdata =>
      {
        var predClass = this.Classify(pdata.Key, classes, threshold);
        var realClass = mapValueToClass(pdata.Value, classes, threshold);
        if (!object.Equals(predClass, realClass))
          lock (errors)
          {
            errors.Add(new ErrorInfo<TObj, Class>(pdata.Key, realClass, predClass));
          }
      });

      return errors;
    }


    private Class mapValueToClass(double value, Class[] classes, double threshold)
    {
      if (classes==null || classes.Length<=0)
        return Class.Unknown;
      if (classes.Length != 2)
        throw new MLException("There must be 2 classes for classification");

      var idx = (value<threshold) ? 0 : 1;
      return classes[idx];
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


    public virtual Class Classify(TObj x, Class[] classes)
    {
      var result = Predict(x);
      return mapValueToClass(result, classes);
    }

    /// <summary>
    /// Returns all errors of the algorithm on some test classified sample
    /// </summary>
    public virtual IEnumerable<ErrorInfo<TObj, Class>> GetClassificationErrors(MultiRegressionSample<TObj> testSample, Class[] classes)
    {
      var errors = new List<ErrorInfo<TObj, Class>>();
      Parallel.ForEach(testSample, pdata =>
      {
        var predClass = this.Classify(pdata.Key, classes);
        var realClass = mapValueToClass(pdata.Value, classes);
        if (!object.Equals(predClass, realClass))
          lock (errors)
          {
            errors.Add(new ErrorInfo<TObj, Class>(pdata.Key, realClass, predClass));
          }
      });

      return errors;
    }


    private Class mapValueToClass(double[] value, Class[] classes)
    {
      if (value==null || classes==null)
        return Class.Unknown;
      if (classes.Length != value.Length)
        throw new MLException("Wrong class count");

      var idx = MathUtils.ArgMax(value);
      return classes[idx];
    }
  }
}
