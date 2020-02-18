%
% superclass for all threshold normalization options
%
% author: Martin F. Schiffner
% date: 2019-08-10
% modified: 2020-02-17
%
classdef threshold < regularization.normalizations.normalization

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        value ( 1, 1 ) double { mustBePositive, mustBeLessThanOrEqual( value, 1 ), mustBeNonempty } = 1	% threshold for normalization of matrix columns

	end % properties

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = threshold( values )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % property validation functions ensure valid values

            %--------------------------------------------------------------
            % 2.) create inactive normalization options
            %--------------------------------------------------------------
            % constructor of superclass
            objects@regularization.normalizations.normalization( size( values ) );

            % iterate threshold normalization options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).value = values( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = threshold( values )

        %------------------------------------------------------------------
        % string array (overload string method)
        %------------------------------------------------------------------
        function strs_out = string( normalizations_threshold )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure class regularization.normalizations.threshold
            if ~isa( normalizations_threshold, 'regularization.normalizations.threshold' )
                errorStruct.message = 'normalizations_threshold must be regularization.normalizations.threshold!';
                errorStruct.identifier = 'string:NoNormalizationThreshold';
                error( errorStruct );
            end

            %--------------------------------------------------------------
            % 2.) create string array
            %--------------------------------------------------------------
            % initialize string array for strs_out
            strs_out = repmat( "", size( normalizations_threshold ) );

            % iterate threshold normalization options
            for index_object = 1:numel( normalizations_threshold )

                strs_out( index_object ) = sprintf( "%s (%6.4f)", 'threshold', normalizations_threshold( index_object ).value );

            end % for index_object = 1:numel( normalizations_threshold )

        end % function strs_out = string( normalizations_threshold )

	end % methods

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods (protected, hidden)
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods (Access = protected, Hidden)

        %------------------------------------------------------------------
        % apply normalization (scalar)
        %------------------------------------------------------------------
        function [ weighting, N_threshold ] = apply_scalar( normalization, weighting )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % calling function ensures class regularization.normalizations.normalization (scalar) for normalization
            % calling function ensures class linear_transforms.weighting (scalar) for weighting

            %--------------------------------------------------------------
            % 2.) apply threshold (scalar)
            %--------------------------------------------------------------
            % apply threshold to weighting matrix
            [ weighting, N_threshold ] = threshold( weighting, normalization.value );

        end % function [ weighting, N_threshold ] = apply_scalar( normalization, weighting )

	end % methods (Access = protected, Hidden)

end % classdef threshold < regularization.normalizations.normalization
