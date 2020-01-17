%
% superclass for all threshold normalization options
%
% author: Martin F. Schiffner
% date: 2019-08-10
% modified: 2020-01-16
%
classdef normalization_threshold < regularization.options.normalization

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        threshold ( 1, 1 ) double { mustBePositive, mustBeLessThanOrEqual( threshold, 1 ), mustBeNonempty } = 1	% threshold for normalization of matrix columns

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = normalization_threshold( thresholds )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid thresholds

            %--------------------------------------------------------------
            % 2.) create inactive normalization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.options.normalization( size( thresholds ) );

            % iterate threshold normalization options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).threshold = thresholds( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = normalization_threshold( thresholds )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( normalizations_threshold )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.options.normalization_threshold
            if ~isa( normalizations_threshold, 'regularization.options.normalization_threshold' )
                errorStruct.message = 'normalizations_threshold must be regularization.options.normalization_threshold!';
                errorStruct.identifier = 'string:NoOptionsNormalizationThreshold';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize string array for strs_out
            strs_out = repmat( "", size( normalizations_threshold ) );

            % iterate threshold normalization options
            for index_object = 1:numel( normalizations_threshold )

                strs_out( index_object ) = sprintf( "%s (%6.4f)", 'threshold', normalizations_threshold( index_object ).threshold );

            end % for index_object = 1:numel( normalizations_threshold )

        end % function strs_out = string( normalizations_threshold )

	end % methods

end % classdef normalization_threshold < regularization.options.normalization
