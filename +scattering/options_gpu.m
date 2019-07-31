%
% abstract superclass for all GPU options
%
% author: Martin F. Schiffner
% date: 2019-07-09
% modified: 2019-07-29
%
classdef (Abstract) options_gpu

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = options_gpu( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'options_gpu:NoRowVector';
                error( errorStruct );
            end

            % ensure positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create GPU options
            %--------------------------------------------------------------
            % repeat GPU options
            objects = repmat( objects, size );

        end % function objects = options_gpu( size )

	end % methods

end % classdef (Abstract) options_gpu
