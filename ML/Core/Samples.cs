using System;
using System.Collections.Generic;
using System.Linq;
using ML.Utils;

namespace ML.Core
{
  public delegate bool SampleMaskDelegate<TObj, TMark>(TObj p, TMark c, int i);

  /// <summary>
  /// Represents a marked (e.g. supplied with corrresponding mark) set of points: [point, mark]
  /// </summary>
  public abstract class MarkedSample<TObj, TMark> : Dictionary<TObj, TMark>
  {
    #region .ctor

      public MarkedSample()
      {
      }

      public MarkedSample(Dictionary<TObj, TMark> other) : base(other)
      {
      }

      public MarkedSample(MarkedSample<TObj, TMark> other) : base(other)
      {
      }

    #endregion

    /// <summary>
    /// All points
    /// </summary>
    public IEnumerable<TObj> Objects { get { return this.Keys; } }

    /// <summary>
    /// All marks
    /// </summary>
    public IEnumerable<TMark> Marks { get { return this.Values.Distinct(); } }
  }

  /// <summary>
  /// Represents a regression (e.g. supplied with corresponding numeric value) set of points: [point, value]
  /// </summary>
  public class RegressionSample<TObj> : MarkedSample<TObj, double>
  {
    #region .ctor

    public RegressionSample()
    {
    }

    public RegressionSample(Dictionary<TObj, double> other) : base(other)
    {
    }

    public RegressionSample(RegressionSample<TObj> other) : base(other)
    {
    }

    #endregion

    /// <summary>
    /// Retrieves subset of the sample
    public RegressionSample<TObj> Subset(int skip, int take)
    {
      return GeneralUtils.Subset<RegressionSample<TObj>, TObj, double>(this, skip, take);
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public IEnumerable<RegressionSample<TObj>> Batch(int size)
    {
      return GeneralUtils.Batch<RegressionSample<TObj>, TObj, double>(this, size);
    }

    /// <summary>
    /// Apply mask to the sample
    /// </summary>
    public RegressionSample<TObj> ApplyMask(SampleMaskDelegate<TObj, double> mask)
    {
      return GeneralUtils.ApplyMask<RegressionSample<TObj>, TObj, double>(this, mask);
    }
  }

  /// <summary>
  /// Represents a multidimensional value (e.g. supplied with numeric vector) set of points: [point, vector]
  /// </summary>
  public class MultiRegressionSample<TObj> : MarkedSample<TObj, double[]>
  {
    #region .ctor

    public MultiRegressionSample()
    {
    }

    public MultiRegressionSample(Dictionary<TObj, double[]> other) : base(other)
    {
    }

    public MultiRegressionSample(MultiRegressionSample<TObj> other) : base(other)
    {
    }

    #endregion

    /// <summary>
    /// Retrieves subset of the sample
    public MultiRegressionSample<TObj> Subset(int skip, int take)
    {
      return GeneralUtils.Subset<MultiRegressionSample<TObj>, TObj, double[]>(this, skip, take);
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public IEnumerable<MultiRegressionSample<TObj>> Batch(int size)
    {
      return GeneralUtils.Batch<MultiRegressionSample<TObj>, TObj, double[]>(this, size);
    }

    /// <summary>
    /// Apply mask to the sample
    /// </summary>
    public MultiRegressionSample<TObj> ApplyMask(SampleMaskDelegate<TObj, double[]> mask)
    {
      return GeneralUtils.ApplyMask<MultiRegressionSample<TObj>, TObj, double[]>(this, mask);
    }
  }

  /// <summary>
  /// Represents a classified (e.g. supplied with corresponding class) set of points: [point, class]
  /// </summary>
  public class ClassifiedSample<TObj> : MarkedSample<TObj, Class>
  {
    private readonly object m_Sync = new object();

    private List<Class> m_CachedClasses;

    #region .ctor

    public ClassifiedSample()
    {
    }

    public ClassifiedSample(Dictionary<TObj, Class> other) : base(other)
    {
    }

    public ClassifiedSample(ClassifiedSample<TObj> other) : base(other)
    {
    }

    #endregion

    public IEnumerable<Class> Classes
    {
      get
      {
        m_CachedClasses = this.Values.Distinct().ToList();
        return m_CachedClasses;
      }
    }

    public IEnumerable<Class> CachedClasses
    {
      get
      {
        if (m_CachedClasses==null)
        {
          lock (m_Sync)
          {
            if (m_CachedClasses==null)
              return Classes;
          }
        }

        return m_CachedClasses;
      }
    }

    /// <summary>
    /// Retrieves subset of the sample
    /// </summary>
    public ClassifiedSample<TObj> Subset(int skip, int take)
    {
      return GeneralUtils.Subset<ClassifiedSample<TObj>, TObj, Class>(this, skip, take);
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public IEnumerable<ClassifiedSample<TObj>> Batch(int size)
    {
      return GeneralUtils.Batch<ClassifiedSample<TObj>, TObj, Class>(this, size);
    }

    /// <summary>
    /// Apply mask to the sample
    /// </summary>
    public ClassifiedSample<TObj> ApplyMask(SampleMaskDelegate<TObj, Class> mask)
    {
      return GeneralUtils.ApplyMask<ClassifiedSample<TObj>, TObj, Class>(this, mask);
    }
  }

}
