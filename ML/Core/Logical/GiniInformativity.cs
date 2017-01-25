using System;
using System.Collections.Generic;
using ML.Contracts;

namespace ML.Core.Logical
{
  public class GiniInformativity : IInformativity
  {
    public Predicate<Point> Max(IEnumerable<Predicate<Point>> patterns, ClassifiedSample sample)
    {
      Predicate<Point> result = null;
      var maxNum = int.MinValue;

      foreach (var pattern in patterns)
      {
        int num = 0;
        var prevs = new List<KeyValuePair<Point, Class>>();

        foreach (var pair in sample)
        {
          if (prevs.Count < 1)
          {
            prevs.Add(new KeyValuePair<Point, Class>(pair.Key, pair.Value));
            continue;
          }

          foreach (var prev in prevs)
          {
            if (pattern(pair.Key) == pattern(prev.Key) && pair.Value.Equals(prev.Value))
              num++;
          }

          prevs.Add(new KeyValuePair<Point, Class>(pair.Key, pair.Value));
        }

        if (num > maxNum)
        {
          maxNum = num;
          result = pattern;
        }
      }

      return result;
    }
  }
}
