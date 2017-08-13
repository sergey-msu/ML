using System;
using ML.Core.Serialization;

namespace ML.Contracts
{
  public interface IMLSerializable
  {
    void Serialize(MLSerializer ser);
    void Deserialize(MLSerializer ser);
  }
}
