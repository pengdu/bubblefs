/*
 * Copyright 2017 LinkedIn Corp. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

// ambry/ambry-api/src/main/java/com.github.ambry/commons/TopicListener.java
// ambry/ambry-api/src/main/java/com.github.ambry/commons/Notifier.java

#ifndef BUBBLEFS_UTILS_SIMPLE_NOTIFIER_H_
#define BUBBLEFS_UTILS_SIMPLE_NOTIFIER_H_

#include <string>

namespace bubblefs {
namespace mysimple {
  
/**
 * A {@code TopicListener} can subscribe topics for messages through a {@link Notifier}.
 * @param <T> The type of message the listener will receive.
 */
template <typename T>
class TopicListener {

  /**
   * After the {@link TopicListener} has subscribed the topic, this method will be called when there is a new message
   * for the topic.
   * @param topic The topic this {@code TopicListener} subscribes.
   * @param message The message for the topic.
   */
 public:
   virtual void onMessage(std::string topic, T message) = 0;
};


/**
 * A {@code Notifier} is a hub that dispatches messages of various topics to {@link TopicListener}s.
 * After a {@link TopicListener} has subscribed to a topic through the {@code Notifier}, a message can
 * be published for that topic, and will be received by the {@link TopicListener}. A topic can be
 * subscribed by multiple {@link TopicListener}, and a {@link TopicListener} can subscribe multiple topics.
 *
 * @param <T> The type of message.
 */
template <typename T>
class Notifier {

  /**
   * Publishes a message for the specified topic. The {@link TopicListener}s that have subscribed to the
   * topic will receive the message.
   * @param topic The topic the message is sent for.
   * @param message The message to send for the topic.
   * @return {@code true} if the message has been sent out successfully, {@code false} otherwise.
   */
 public:
  virtual bool publish(std::string topic, T message) = 0;

  /**
   * Let a {@link TopicListener} subscribe to a topic. After subscription, it will receive the messages
   * published for the topic.
   * @param topic The topic to subscribe.
   * @param listener The {@link TopicListener} who subscribes the topic.
   */
  virtual void subscribe(std::string topic, TopicListener<T> listener) = 0;

  /**
   * Let a {@link TopicListener} unsubscribe from a topic, so it will no longer receive the messages for
   * the topic.
   * @param topic The topic to unsubscribe.
   * @param listener The {@link TopicListener} who unsubscribes the topic.
   */
  virtual void unsubscribe(std::string topic, TopicListener<T> listener) = 0;
};

} // namespace mysimple
} // namespace bubblefs

#endif // BUBBLEFS_UTILS_SIMPLE_NOTIFIER_H_