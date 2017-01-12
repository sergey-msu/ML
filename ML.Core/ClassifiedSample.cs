using System;
using System.Collections.Generic;
using System.Linq;

namespace ML.Core
{
  public class ClassifiedSample : Dictionary<Point, Class>
  {
    public ClassifiedSample()
    {
    }

    public ClassifiedSample(ClassifiedSample other) : base(other)
    {
    }

    public IEnumerable<Point> Points  { get { return this.Keys; } }
    public IEnumerable<Class> Classes { get { return this.Values.Distinct(); } }
  }
}
