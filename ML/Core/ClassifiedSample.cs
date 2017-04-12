using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Core
{
  /// <summary>
  /// Represents a classified (e.g. supplied with corrresponding class) set of points: [point, class]
  /// </summary>
  public class ClassifiedSample<TObj> : Dictionary<TObj, Class>
  {
    public ClassifiedSample()
    {
    }

    public ClassifiedSample(Dictionary<TObj, Class> other) : base(other)
    {
    }

    public ClassifiedSample(ClassifiedSample<TObj> other) : base(other)
    {
    }

    /// <summary>
    /// All points
    /// </summary>
    public IEnumerable<TObj> Objects { get { return this.Keys; } }

    /// <summary>
    /// All classes
    /// </summary>
    public IEnumerable<Class> Classes { get { return this.Values.Distinct(); } }

    /// <summary>
    /// Retrieves subset of the sample
    public ClassifiedSample<TObj> Subset(int skip, int take)
    {
      if (skip<0) throw new MLException("Skip value must be non-negative");
      if (take<=0) throw new MLException("Take value must be positive");

      var result = new ClassifiedSample<TObj>();
      foreach (var item in this.Skip(skip).Take(take))
        result[item.Key] = item.Value;

      return result;
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public IEnumerable<ClassifiedSample<TObj>> Batch(int size)
    {
      if (size <= 0)
        throw new MLException("Batch size must be positive");

      var bucket = new ClassifiedSample<TObj>();
      var count = 0;

      foreach (var item in this)
      {
        bucket[item.Key] = item.Value;
        count++;
        if (count != size) continue;

        yield return bucket;

        bucket.Clear();
        count = 0;
      }

      if (bucket.Count==0) yield break;

      yield return bucket;
    }
  }
}
