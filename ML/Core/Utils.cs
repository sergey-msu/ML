using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Core
{
  public static class Utils
  {
    public static TResult GetThroughMap<TKey, TResult>(TKey key, Dictionary<TKey, TResult> map)
    {
      TResult result;
      if (!map.TryGetValue(key, out result))
      {
        result = (TResult)Activator.CreateInstance(typeof(TResult), key);
        map[key] = result;
      }

      return result;
    }

    /// <summary>
    /// Retrieves subset of the sample
    public static TSample Subset<TSample, TObj, TMark>(TSample src, int skip, int take)
      where TSample : MarkedSample<TObj, TMark>, new()
    {
      if (src==null) throw new MLException("Source sample is null");
      if (skip<0)    throw new MLException("Skip value must be non-negative");
      if (take<=0)   throw new MLException("Take value must be positive");

      var result = new TSample();
      foreach (var item in src.Skip(skip).Take(take))
        result[item.Key] = item.Value;

      return result;
    }

    /// <summary>
    /// Enumerate sample batches
    /// </summary>
    public static IEnumerable<TSample> Batch<TSample, TObj, TMark>(TSample src, int size)
      where TSample : MarkedSample<TObj, TMark>, new()
    {
      if (src==null) throw new MLException("Source sample is null");
      if (size <= 0) throw new MLException("Batch size must be positive");

      var bucket = new TSample();
      var count = 0;

      foreach (var item in src)
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

    public static TSample ApplyMask<TSample, TObj, TMark>(TSample src, SampleMaskDelegate<TObj, TMark> mask)
      where TSample : MarkedSample<TObj, TMark>, new()
    {
      if (src==null) throw new MLException("Source sample is null");

      if (mask==null) return src;

      var result = new TSample();
      int counter = -1;
      foreach (var item in src)
      {
        counter++;
        if (mask(item.Key, item.Value, counter))
          result[item.Key] = item.Value;
      }

      return result;
    }

    public static MultiRegressionSample<TObj> ClassifiedToRegressionSample<TObj>(ClassifiedSample<TObj> sample)
    {
      if (sample==null) return null;

      var classes = sample.Classes.ToList();
      var count = classes.Count;
      var marks = new double[count][];
      for (int i=0; i<count; i++)
      {
        marks[i] = new double[count];
        marks[i][i] = 1;
      }

      var result = new MultiRegressionSample<TObj>();
      foreach (var data in sample)
      {
        var cls = data.Value;
        var idx = classes.IndexOf(cls);
        var mark = marks[idx];
        result.Add(data.Key, mark);
      }

      return result;
    }

  }
}
