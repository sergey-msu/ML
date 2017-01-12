using System.Collections.Generic;

namespace ML.Core.Contracts
{
  public interface IAlgorithm
  {
    string ID { get; }

    string Name { get; }

    ClassifiedSample TrainingSample { get; }

    Dictionary<string, Class> Classes { get; }

    Class Classify(Point point);

    float EstimateClose(Point point, Class cls);
  }
}
