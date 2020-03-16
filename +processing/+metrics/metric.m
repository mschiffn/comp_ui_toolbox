%
% superclass for all image quality metrics
%
% author: Martin F. Schiffner
% date: 2020-02-29
% modified: 2020-02-29
%
classdef (Abstract) metric

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = metric( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'metric:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create image quality metrics
            %--------------------------------------------------------------
            % repeat default image quality metric
            objects = repmat( objects, size );

        end % function objects = metric( size )

        %------------------------------------------------------------------
        % evaluate metric
        %------------------------------------------------------------------
        function results = evaluate( metrics, images )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class processing.metrics.metric
            if ~isa( metrics, 'processing.metrics.metric' )
                errorStruct.message = 'metrics must be processing.metrics.metric!';
                errorStruct.identifier = 'evaluate:NoMetrics';
                error( errorStruct );
            end

            % ensure class processing.image
            if ~isa( images, 'processing.image' )
                errorStruct.message = 'images must be processing.image!';
                errorStruct.identifier = 'evaluate:NoImages';
                error( errorStruct );
            end

            % multiple metrics / single images
            if ~isscalar( metrics ) && isscalar( images )
                images = repmat( images, size( metrics ) );
            end

            % single metrics / multiple images
            if isscalar( metrics ) && ~isscalar( images )
                metrics = repmat( metrics, size( images ) );
            end

            % ensure equal number of dimensions and sizes
            auxiliary.mustBeEqualSize( metrics, images );

            %--------------------------------------------------------------
            % 2.) evaluate metrics
            %--------------------------------------------------------------
            % specify cell array for results
            results = cell( size( images ) );

            % iterate image quality metrics
            for index_metric = 1:numel( metrics )

                %----------------------------------------------------------
                % b) evaluate metric (scalar)
                %----------------------------------------------------------
                results{ index_metric } = evaluate_scalar( metrics( index_metric ), images( index_metric ) );

            end % for index_metric = 1:numel( metrics )

            % TODO: merge metrics into array
            N_results = cellfun( @numel, results );
            if all( N_results( : ) == N_results( 1 ) )
                results = reshape( cat( 1, results{ : } ), [ size( images ), N_results( 1 ) ] );
            end

            % avoid cell array for single images
            if isscalar( images )
                results = results{ 1 };
            end

        end % function results = evaluate( metrics, images )

	end % methods

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (abstract, protected, and hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Abstract, Access = protected, Hidden)

        %------------------------------------------------------------------
        % evaluate metric (scalar)
        %------------------------------------------------------------------
        results = evaluate_scalar( metric, image )

	end % methods (Abstract, Access = protected, Hidden)

end % classdef (Abstract) metric
