%
% template for all abstract superclasses
%
% author: Martin F. Schiffner
% date: 2020-07-22
% modified: 2020-07-22
%
classdef (Abstract) abstract_class_template

	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	%% methods
	%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
	methods

        %------------------------------------------------------------------
        % constructor
        %------------------------------------------------------------------
        function objects = abstract_class_template( size )

            %--------------------------------------------------------------
            % 1.) check arguments
            %--------------------------------------------------------------
            % ensure one argument
            narginchk( 1, 1 );

            % ensure row vector for size
            if ~isrow( size )
                errorStruct.message = 'size must be a row vector!';
                errorStruct.identifier = 'abstract_class_template:NoRowVector';
                error( errorStruct );
            end

            % ensure nonempty positive integers
            mustBePositive( size );
            mustBeInteger( size );
            mustBeNonempty( size );

            %--------------------------------------------------------------
            % 2.) create scattering options
            %--------------------------------------------------------------
            % repeat default regularization options
            objects = repmat( objects, size );

        end % function objects = abstract_class_template( size )

	end % methods

end % classdef (Abstract) abstract_class_template
