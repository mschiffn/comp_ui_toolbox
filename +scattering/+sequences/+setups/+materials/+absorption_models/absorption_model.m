%
% superclass for all absorption models
%
% author: Martin F. Schiffner
% date: 2016-08-25
% modified: 2019-06-05
%
classdef absorption_model

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% properties
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	properties (SetAccess = private)

        % independent properties
        str_name                    % name of absorption model

    end % properties

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods (concrete)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = absorption_model( strs_name )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure cell array for strs_name
            if ~iscell( strs_name )
                strs_name = { strs_name };
            end

            %--------------------------------------------------------------
            % 2.) create absorption models
            %--------------------------------------------------------------
            % repeat default absorption model
            objects = repmat( objects, size( strs_name ) );

            % iterate absorption models
            for index_object = 1:numel( objects )

                % internal properties
                objects( index_object ).str_name = strs_name{ index_object };

            end % for index_object = 1:numel( objects )

        end % function objects = absorption_model( strs_name )

    end % methods

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% methods (abstract)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods (Abstract)

        axes_k_tilde = compute_wavenumbers( time_causal, axes_f )

    end

end % classdef (Abstract) absorption_model
