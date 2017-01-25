using System;
using System.Runtime.Serialization;

namespace ML.Core
{
  /// <summary>
  /// Base exception thrown by the ML library
  /// </summary>
  [Serializable]
  public class MLException : Exception
  {
    public MLException()
    {
    }

    public MLException(string message)
      : base(message)
    {
    }

    public MLException(string message, Exception inner)
      : base(message, inner)
    {
    }

    protected MLException(SerializationInfo info, StreamingContext context)
      : base(info, context)
    {
    }
  }
}
