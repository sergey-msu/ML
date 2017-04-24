using System;
using System.Collections.Generic;

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

  }
}
