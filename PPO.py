作者：何枝
链接：https://www.zhihu.com/question/651021172/answer/3513159005
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

def compute_loss(self, inputs):
    prompts = inputs["prompts"]
    log_probs = inputs["logprobs"]
    ref_log_probs = inputs["ref_logprobs"]
    reward_score = inputs["rewards"]
    baseline_reward_score = inputs["baseline_rewards"]
    attention_mask = inputs["attention_mask"]
    seq = inputs["input_ids"]

    start = prompts.size()[-1] - 1
    action_mask = attention_mask[:, 1:]

    with torch.no_grad():
        kl_divergence = -(log_probs - ref_log_probs)
        kl_divergence = self.kl_ctl * kl_divergence

        reward_score = reward_score - baseline_reward_score         # 真实 reward
        returns, kl_ratio = self.compute_returns(
            prompts, kl_divergence, reward_score, action_mask
        )

    # process the new outputs
    batch = {"input_ids": seq, "attention_mask": attention_mask}
    logits = self.actor_model(**batch, use_cache=False).logits
    log_probs = gather_log_probs(logits[:, :-1, :], seq[:, 1:])

    actor_loss = self.actor_loss_fn(
        log_probs[:, start:], returns[:, start:], action_mask[:, start:]
    )
    return actor_loss, returns[:, start:], kl_ratio


# reward & basline_reward_score 计算如下:
seq = self._generate_sequence(
    self.actor_model,
    prompts,
    ...
)
baseline_seq = self._generate_sequence(
    self.actor_model,
    prompts,
    ...
    do_sample=False,
)
reward_score = self.reward_model.forward_value(
    seq, action_mask, prompt_length=self.prompt_length
)
baseline_reward_score = self.reward_model.forward_value(
    baseline_seq, baseline_action_mask, prompt_length=self.prompt_length
)
