using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using ML.Core;

namespace ML.Utils
{
  public static class GeneralUtils
  {
    private static Dictionary<int, int> s_SampleDimCache = new Dictionary<int, int>();


    public static int GetDimension(this ClassifiedSample<double[]> sample, bool cache = true)
    {
      return sample.First().Key.Length;
    }

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

    /// <summary>
    /// Calculates h(z) = -z*log2(z)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double EntropyH(double z)
    {
      return (0.0D <= z && z < double.Epsilon) ? 0.0D : -z*Math.Log(z)*MathConsts.ENTROPY_COEFF;
    }

    /// <summary>
    /// Calculates maximum value within array alog with its index
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static int ArgMax<T>(T[] array)
      where T : IComparable<T>
    {
      var idx = -1;
      var max = default(T);

      if (array==null) return -1;

      var len = array.Length;
      for (int i=0; i<len; i++)
      {
        var val = array[i];
        if (i==0 || val.CompareTo(max)>0)
        {
          idx = i;
          max = val;
        }
      }

      return idx;
    }

    /// <summary>
    /// Throws if arrays have different lenghts
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void CheckDimensions(double[] p1, double[] p2)
    {
      if (p1.Length != p2.Length)
        throw new MLException("Can not add point with different dimension");
    }

    /// <summary>
    /// Calculates power of a given number.
    /// Works 10-times falser than Math.Pow (Release x64)
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static double Pow(double p, int pow)
    {
      if (p==0)
      {
        if (pow <= 0) return double.NaN; // throw new Exception("Zero raised in non-positive power");
        return 0;
      }

      if (pow==0) return 1;
      if (pow==1) return p;

      if (pow<0)
      {
        pow = -pow;
        p = 1.0D/p;
      }

      var result = p;
      for (int i=0; i<pow-1; i++)
        result *= p;

      return result;
    }
  }
}
