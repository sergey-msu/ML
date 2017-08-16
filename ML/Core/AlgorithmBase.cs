using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading.Tasks;
using ML.Contracts;
using ML.Utils;
using ML.Core.Serialization;
using System.IO.Compression;

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

    #region Serialization

    public void Serialize(Stream stream)
    {
      using (var ims = new MemoryStream())
      using (var writer = new StreamWriter(ims) { AutoFlush=true })
      {
        var ser = new MLSerializer(writer);

        Serialize(ser);

        ims.Position = 0;
        using (var compr = new GZipStream(stream, CompressionMode.Compress))
          ims.CopyTo(compr);
      }
    }

    public void Deserialize(Stream stream)
    {
      using (var compr = new GZipStream(stream, CompressionMode.Decompress))
      using (var ims = new MemoryStream())
      {
        compr.CopyTo(ims);
        ims.Position = 0;

        using (var reader = new StreamReader(ims))
        {
          var ser = new MLSerializer(reader);
          Deserialize(ser);
        }
      }
    }

    public virtual void Serialize(MLSerializer ser)
    {
      ser.Write("TYPE", GetType().FullName);
    }

    public virtual void Deserialize(MLSerializer ser)
    {
      var type = ser.ReadString("TYPE");
      if (!string.Equals(type, GetType().FullName, StringComparison.InvariantCultureIgnoreCase))
        throw new MLException(string.Format("Wrong type: expected {0}, actual {1}", type ?? "NULL", GetType().FullName));
    }

    #endregion
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


    public Class[] Classes { get; private set; }

    /// <summary>
    /// Make a prediction
    /// </summary>
    public override Class Predict(TObj obj)
    {
      var tokens = PredictTokens(obj, 1);
      if (tokens.Length <= 0) return Class.Unknown;

      return tokens[0].Class;
    }

    /// <summary>
    /// Returns set of marks (i.e. tags) for the specified object
    /// </summary>
    public abstract ClassScore[] PredictTokens(TObj obj, int cnt);

    protected override void DoTrain()
    {
      Classes = TrainingSample.Classes.ToArray();
    }

    protected override double CalculateProximity(Class mark1, Class mark2, double threshold)
    {
      return (mark1.Equals(mark2)) ? 0 : 1;
    }

    #region Serialization

    public override void Serialize(MLSerializer ser)
    {
      base.Serialize(ser);

      ser.Write("CLASSES", Classes.Select(c => c.Name));
    }

    public override void Deserialize(MLSerializer ser)
    {
      base.Deserialize(ser);

      var classes = new List<Class>();
      var names = ser.ReadStrings("CLASSES");
      var idx = 0;

      foreach (var name in names)
      {
        var cls = new Class(name, idx++);
        classes.Add(cls);
      }

      Classes = classes.ToArray();
    }

    #endregion
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

  public struct ClassScore : IEquatable<ClassScore>
  {
    public ClassScore(Class cls, double score)
    {
      Class = cls;
      Score = score;
    }

    public readonly Class Class;
    public readonly double Score;


    public bool Equals(ClassScore other)
    {
      return Class.Equals(other.Class) && Score==other.Score;
    }

    public override bool Equals(object obj)
    {
      if (!(obj is ClassScore)) return false;
      return Equals((ClassScore)obj);
    }

    public override int GetHashCode()
    {
      return Class.GetHashCode() ^ Score.GetHashCode();
    }
  }
}
