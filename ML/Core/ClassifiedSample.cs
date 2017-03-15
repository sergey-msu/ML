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
  }
}
