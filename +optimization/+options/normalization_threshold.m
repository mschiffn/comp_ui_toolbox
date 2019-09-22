%
% superclass for all threshold normalization options
%
% author: Martin F. Schiffner
% date: 2019-08-10
% modified: 2019-09-22
%
classdef normalization_threshold < optimization.options.normalization

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
            objects@optimization.options.normalization( size( thresholds ) );

            % iterate threshold normalization options
            for index_object = 1:numel( objects )

                % set independent properties
                objects( index_object ).threshold = thresholds( index_object );

            end % for index_object = 1:numel( objects )

        end % function objects = normalization_threshold( thresholds )

	end % methods

end % classdef normalization_threshold < optimization.options.normalization
