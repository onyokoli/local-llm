#include "types.h"

namespace localllm {

// ModelInfo implementation
json ModelInfo::to_json() const {
    json j = {
        {"id", id},
        {"name", name},
        {"base_path", base_path},
        {"architecture_type", architecture_type},
        {"parameter_count", parameter_count},
        {"context_length", context_length},
        {"is_quantized", is_quantized},
        {"quantization_method", quantization_method},
        {"is_finetuned", is_finetuned}
    };

    if (parent_model_id) {
        j["parent_model_id"] = *parent_model_id;
    }

    if (is_finetuned) {
        j["finetuning_method"] = finetuning_method;
        j["lora_rank"] = lora_rank;
        j["training_dataset"] = training_dataset;
        j["training_date"] = training_date;
        j["training_steps"] = training_steps;
        j["final_loss"] = final_loss;
    }

    return j;
}

ModelInfo ModelInfo::from_json(const json& j) {
    ModelInfo info;
    info.id = j["id"];
    info.name = j["name"];
    info.base_path = j["base_path"];
    info.architecture_type = j["architecture_type"];
    info.parameter_count = j["parameter_count"];
    info.context_length = j["context_length"];
    info.is_quantized = j["is_quantized"];
    info.quantization_method = j["quantization_method"];
    info.is_finetuned = j["is_finetuned"];

    if (j.contains("parent_model_id")) {
        info.parent_model_id = j["parent_model_id"];
    }

    if (info.is_finetuned) {
        info.finetuning_method = j["finetuning_method"];
        info.lora_rank = j["lora_rank"];
        info.training_dataset = j["training_dataset"];
        info.training_date = j["training_date"];
        info.training_steps = j["training_steps"];
        info.final_loss = j["final_loss"];
    }

    return info;
}

// DatasetInfo implementation
json DatasetInfo::to_json() const {
    return {
        {"id", id},
        {"name", name},
        {"data_path", data_path.string()},
        {"num_examples", num_examples},
        {"total_tokens", total_tokens},
        {"format_type", format_type},
        {"created_date", created_date},
        {"is_processed", is_processed},
        {"format_config", format_config},
        {"val_split_ratio", val_split_ratio},
        {"train_examples", train_examples},
        {"val_examples", val_examples},
        {"input_template", input_template},
        {"output_template", output_template},
        {"avg_input_length", avg_input_length},
        {"avg_output_length", avg_output_length},
        {"max_length", max_length}
    };
}

DatasetInfo DatasetInfo::from_json(const json& j) {
    DatasetInfo info;
    info.id = j["id"];
    info.name = j["name"];
    info.data_path = j["data_path"];
    info.num_examples = j["num_examples"];
    info.total_tokens = j["total_tokens"];
    info.format_type = j["format_type"];
    info.created_date = j["created_date"];
    info.is_processed = j["is_processed"];
    info.format_config = j["format_config"];
    info.val_split_ratio = j["val_split_ratio"];
    info.train_examples = j["train_examples"];
    info.val_examples = j["val_examples"];
    info.input_template = j["input_template"];
    info.output_template = j["output_template"];
    info.avg_input_length = j["avg_input_length"];
    info.avg_output_length = j["avg_output_length"];
    info.max_length = j["max_length"];
    return info;
}

// FineTuneConfig implementation
json FineTuneConfig::to_json() const {
    std::string method_str;
    switch (method) {
        case Method::FULL_FINETUNE: method_str = "full"; break;
        case Method::LORA: method_str = "lora"; break;
        case Method::QLORA: method_str = "qlora"; break;
    }

    return {
        {"learning_rate", learning_rate},
        {"batch_size", batch_size},
        {"epochs", epochs},
        {"weight_decay", weight_decay},
        {"adam_beta1", adam_beta1},
        {"adam_beta2", adam_beta2},
        {"adam_epsilon", adam_epsilon},
        {"method", method_str},
        {"lora_rank", lora_rank},
        {"lora_alpha", lora_alpha},
        {"lora_dropout", lora_dropout},
        {"lora_target_modules", lora_target_modules},
        {"quantization_type", quantization_type},
        {"max_memory_usage", max_memory_usage},
        {"num_threads", num_threads},
        {"use_gpu", use_gpu},
        {"enable_checkpointing", enable_checkpointing},
        {"checkpoint_steps", checkpoint_steps},
        {"checkpoint_dir", checkpoint_dir},
        {"gradient_accumulation_steps", gradient_accumulation_steps},
        {"warmup_steps", warmup_steps},
        {"max_grad_norm", max_grad_norm},
        {"context_length", context_length},
        {"eval_during_training", eval_during_training},
        {"eval_steps", eval_steps},
        {"logging_steps", logging_steps}
    };
}

FineTuneConfig FineTuneConfig::from_json(const json& j) {
    FineTuneConfig config;

    config.learning_rate = j.value("learning_rate", DEFAULT_LEARNING_RATE);
    config.batch_size = j.value("batch_size", DEFAULT_BATCH_SIZE);
    config.epochs = j.value("epochs", DEFAULT_EPOCHS);
    config.weight_decay = j.value("weight_decay", DEFAULT_WEIGHT_DECAY);
    config.adam_beta1 = j.value("adam_beta1", DEFAULT_ADAM_BETA1);
    config.adam_beta2 = j.value("adam_beta2", DEFAULT_ADAM_BETA2);
    config.adam_epsilon = j.value("adam_epsilon", DEFAULT_ADAM_EPSILON);

    std::string method_str = j.value("method", "lora");
    if (method_str == "full") {
        config.method = Method::FULL_FINETUNE;
    } else if (method_str == "lora") {
        config.method = Method::LORA;
    } else if (method_str == "qlora") {
        config.method = Method::QLORA;
    }

    config.lora_rank = j.value("lora_rank", DEFAULT_LORA_RANK);
    config.lora_alpha = j.value("lora_alpha", DEFAULT_LORA_ALPHA);
    config.lora_dropout = j.value("lora_dropout", 0.05f);

    if (j.contains("lora_target_modules")) {
        config.lora_target_modules = j["lora_target_modules"].get<std::vector<std::string>>();
    }

    config.quantization_type = j.value("quantization_type", "q4_0");
    config.max_memory_usage = j.value("max_memory_usage", 0ULL);
    config.num_threads = j.value("num_threads", 0);
    config.use_gpu = j.value("use_gpu", true);
    config.enable_checkpointing = j.value("enable_checkpointing", true);
    config.checkpoint_steps = j.value("checkpoint_steps", 100);
    config.checkpoint_dir = j.value("checkpoint_dir", "");
    config.gradient_accumulation_steps = j.value("gradient_accumulation_steps", 1);
    config.warmup_steps = j.value("warmup_steps", 100);
    config.max_grad_norm = j.value("max_grad_norm", 1.0f);
    config.context_length = j.value("context_length", DEFAULT_CONTEXT_LENGTH);
    config.eval_during_training = j.value("eval_during_training", true);
    config.eval_steps = j.value("eval_steps", 50);
    config.logging_steps = j.value("logging_steps", 10);

    return config;
}

// InferenceConfig implementation
json InferenceConfig::to_json() const {
    return {
        {"temperature", temperature},
        {"max_tokens", max_tokens},
        {"top_p", top_p},
        {"top_k", top_k},
        {"repetition_penalty", repetition_penalty},
        {"n_batch", n_batch},
        {"context_length", context_length},
        {"num_threads", num_threads},
        {"use_gpu", use_gpu},
        {"stream", stream},
        {"seed", seed}
    };
}

InferenceConfig InferenceConfig::from_json(const json& j) {
    InferenceConfig config;

    config.temperature = j.value("temperature", 0.8f);
    config.max_tokens = j.value("max_tokens", 512);
    config.top_p = j.value("top_p", 0.95f);
    config.top_k = j.value("top_k", 40);
    config.repetition_penalty = j.value("repetition_penalty", 1.1f);
    config.n_batch = j.value("n_batch", 512);
    config.context_length = j.value("context_length", DEFAULT_CONTEXT_LENGTH);
    config.num_threads = j.value("num_threads", 0);
    config.use_gpu = j.value("use_gpu", true);
    config.stream = j.value("stream", false);
    config.seed = j.value("seed", 0ULL);

    return config;
}

} // namespace localllm
