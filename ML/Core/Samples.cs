using System;
using System.Collections.Generic;
using System.Linq;

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
      return Utils.Subset<RegressionSample<TObj>, TObj, double>(this, skip, take);
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public IEnumerable<RegressionSample<TObj>> Batch(int size)
    {
      return Utils.Batch<RegressionSample<TObj>, TObj, double>(this, size);
    }

    /// <summary>
    /// Apply mask to the sample
    /// </summary>
    public RegressionSample<TObj> ApplyMask(SampleMaskDelegate<TObj, double> mask)
    {
      return Utils.ApplyMask<RegressionSample<TObj>, TObj, double>(this, mask);
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
      return Utils.Subset<MultiRegressionSample<TObj>, TObj, double[]>(this, skip, take);
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public IEnumerable<MultiRegressionSample<TObj>> Batch(int size)
    {
      return Utils.Batch<MultiRegressionSample<TObj>, TObj, double[]>(this, size);
    }

    /// <summary>
    /// Apply mask to the sample
    /// </summary>
    public MultiRegressionSample<TObj> ApplyMask(SampleMaskDelegate<TObj, double[]> mask)
    {
      return Utils.ApplyMask<MultiRegressionSample<TObj>, TObj, double[]>(this, mask);
    }
  }

  /// <summary>
  /// Represents a classified (e.g. supplied with corresponding class) set of points: [point, class]
  /// </summary>
  public class ClassifiedSample<TObj> : MarkedSample<TObj, Class>
  {
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

    public IEnumerable<Class> Classes { get { return this.Values.Distinct(); } }

    /// <summary>
    /// Retrieves subset of the sample
    /// </summary>
    public ClassifiedSample<TObj> Subset(int skip, int take)
    {
      return Utils.Subset<ClassifiedSample<TObj>, TObj, Class>(this, skip, take);
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public IEnumerable<ClassifiedSample<TObj>> Batch(int size)
    {
      return Utils.Batch<ClassifiedSample<TObj>, TObj, Class>(this, size);
    }

    /// <summary>
    /// Apply mask to the sample
    /// </summary>
    public ClassifiedSample<TObj> ApplyMask(SampleMaskDelegate<TObj, Class> mask)
    {
      return Utils.ApplyMask<ClassifiedSample<TObj>, TObj, Class>(this, mask);
    }
  }

  ///// <summary>
  ///// Represents a classified (e.g. supplied with corresponding class) set of points: [point, class]
  ///// </summary>
  //public class ClassifiedSample<TObj> : MultiRegressionSample<TObj>
  //{
  //  private Dictionary<Class, double[]> m_ClassMapping;

  //  #region .ctor

  //  public ClassifiedSample(IEnumerable<Class> classes)
  //  {
  //    ctor(classes);
  //  }

  //  public ClassifiedSample(IEnumerable<Class> classes, Dictionary<TObj, double[]> other) : base(other)
  //  {
  //    ctor(classes);
  //  }

  //  public ClassifiedSample(IEnumerable<Class> classes, ClassifiedSample<TObj> other) : base(other)
  //  {
  //    ctor(classes);
  //  }

  //  private void ctor(IEnumerable<Class> classes)
  //  {
  //    if (classes==null || !classes.Any())
  //      throw new MLException("ClassifiedSample.ctor(classes=null|empty)");

  //    var clist = classes.ToList();
  //    m_ClassMapping = new Dictionary<Class, double[]>();

  //    int idx = 0;
  //    foreach (var cls in clist)
  //    {
  //      var value = new double[clist.Count];
  //      value[idx] = 1;
  //      m_ClassMapping[cls] = value;
  //    }
  //  }

  //  #endregion

  //  public IEnumerable<Class> Classes { get { return m_ClassMapping.Keys; } }

  //  public void Add(TObj obj, Class cls)
  //  {
  //    double[] value;
  //    if (!m_ClassMapping.TryGetValue(cls, out value))
  //      new MLException("Unknown class");

  //    this.Add(obj, value);
  //  }

  //  public double[] MarkFor(Class cls)
  //  {
  //    return m_ClassMapping[cls];
  //  }

  //  public Class ClassFor(TObj obj)
  //  {
  //    if (obj==null)
  //      throw new MLException("Object can not be null");

  //    double[] mark;
  //    if (!this.TryGetValue(obj, out mark))
  //      throw new MLException("Unknown object");
  //    if (!m_ClassMapping.ContainsValue(mark))
  //      throw new MLException("Unknown value mark");

  //    return m_ClassMapping.First(m => m.Value==mark).Key;
  //  }

  //  public Class ClassFor(double[] mark)
  //  {
  //    if (mark==null)
  //      throw new MLException("Mark can not be null");
  //    if (!m_ClassMapping.ContainsValue(mark))
  //      throw new MLException("Unknown value mark");

  //    return m_ClassMapping.First(m => m.Value==mark).Key;
  //  }

  //  /// <summary>
  //  /// Retrieves subset of the sample
  //  /// </summary>
  //  public ClassifiedSample<TObj> Subset(int skip, int take)
  //  {
  //    return Utils.Subset<ClassifiedSample<TObj>, TObj, double[]>(this, skip, take, Classes);
  //  }

  //  /// <summary>
  //  /// Enumerate sample batches
  //  /// </summary>
  //  public new IEnumerable<ClassifiedSample<TObj>> Batch(int size)
  //  {
  //    return Utils.Batch<ClassifiedSample<TObj>, TObj, double[]>(this, size, Classes);
  //  }

  //  /// <summary>
  //  /// Apply mask to the sample
  //  /// </summary>
  //  public new ClassifiedSample<TObj> ApplyMask(SampleMaskDelegate<TObj, double[]> mask)
  //  {
  //    return Utils.ApplyMask<ClassifiedSample<TObj>, TObj, double[]>(this, mask, Classes);
  //  }
  //}

}
