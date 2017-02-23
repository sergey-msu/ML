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

  /// <summary>
  /// Exception thrown when computing network/layer index was not properly set up
  /// </summary>
  [Serializable]
  public class MLCorruptedIndexException : Exception
  {
    public MLCorruptedIndexException()
    {
    }

    public MLCorruptedIndexException(string message)
      : base(message)
    {
    }

    public MLCorruptedIndexException(string message, Exception inner)
      : base(message, inner)
    {
    }

    protected MLCorruptedIndexException(SerializationInfo info, StreamingContext context)
      : base(info, context)
    {
    }
  }
}
